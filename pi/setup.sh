#!/usr/bin/env bash
# MaxiBox auf dem Raspberry Pi einrichten (einmalig, kann gefahrlos wiederholt werden).
# Aufruf auf dem Pi:  bash ~/maxibox/pi/setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_DIR/venv"
ENV_FILE="$REPO_DIR/.env"
RUN_USER="$(id -un)"

step() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }

# Wert in .env lesen bzw. setzen; die Datei bleibt nur für den Benutzer lesbar
get_env() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | head -n1 | cut -d= -f2- || true; }
set_env() {
    local tmp
    tmp="$(mktemp)"
    grep -vE "^$1=" "$ENV_FILE" > "$tmp" 2>/dev/null || true
    printf '%s=%s\n' "$1" "$2" >> "$tmp"
    cat "$tmp" > "$ENV_FILE"
    rm -f "$tmp"
}
# Karten-IDs aus "arecord -l" / "aplay -l", z. B. "card 2: Device [USB Audio] ..." -> Device
card_ids() { sed -n 's/^card [0-9]*: \([^ ]*\) .*/\1/p'; }

step "1/6 安装软件包（需要输入树莓派的密码）"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip alsa-utils mpg123 curl git python3-gpiozero python3-lgpio

step "2/6 寻找 USB 麦克风和喇叭"
# Der Pi 5 hat kein eingebautes Mikrofon; seine eigenen Ausgänge heissen vc4hdmi*
mic_card="$(arecord -l 2>/dev/null | card_ids | head -n1 || true)"
spk_card="$(aplay -l 2>/dev/null | card_ids | grep -v '^vc4hdmi' | head -n1 || true)"
if [ -z "$mic_card" ] || [ -z "$spk_card" ]; then
    warn "没找到麦克风或喇叭。请检查 USB 线，然后重新运行本脚本。"
    arecord -l || true
    aplay -l || true
    exit 1
fi
MIC_DEVICE="plughw:CARD=$mic_card,DEV=0"
SPEAKER_DEVICE="plughw:CARD=$spk_card,DEV=0"
echo "麦克风：$MIC_DEVICE"
echo "喇叭：  $SPEAKER_DEVICE"
# Lautstärke und Mikrofonpegel setzen; die Reglernamen hängen vom Gerät ab
for ctl in PCM Speaker Master Headphone; do
    amixer -q -c "$spk_card" sset "$ctl" 90% unmute 2>/dev/null && break
done || true
for ctl in Mic Capture; do
    amixer -q -c "$mic_card" sset "$ctl" 80% cap 2>/dev/null && break
done || true

step "3/6 录音测试"
test_wav="$(mktemp --suffix=.wav)"
read -rp "准备好后按 Enter，然后对着麦克风说一句话（4 秒）" _
if arecord -q -D "$MIC_DEVICE" -f S16_LE -r 16000 -c 1 -d 4 "$test_wav"; then
    echo "播放刚才的录音……"
    aplay -q -D "$SPEAKER_DEVICE" "$test_wav" || warn "播放失败。"
else
    warn "录音失败。"
fi
rm -f "$test_wav"
read -rp "听到自己的声音了吗？[y/n] " heard
if [ "${heard:-n}" != "y" ]; then
    warn "先继续。之后可以用 alsamixer -c $spk_card 调喇叭音量，用 alsamixer -c $mic_card 调麦克风（按 F4 切到录音）。"
fi

step "4/6 安装 Python 环境（需要一两分钟）"
# gpiozero und lgpio (für den Knopf am Pi 5) kommen aus apt, deshalb --system-site-packages
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$REPO_DIR/requirements.txt"

step "5/6 API 密钥（只保存在这台树莓派的 $ENV_FILE 里）"
(umask 077 && touch "$ENV_FILE")
chmod 600 "$ENV_FILE"
if [ -z "$(get_env GROQ_API_KEY)" ]; then
    read -rsp "GROQ_API_KEY（Render 的 Environment 页面里有）：" value
    echo
    set_env GROQ_API_KEY "$value"
fi
if [ -z "$(get_env ANTHROPIC_API_KEY)" ]; then
    read -rsp "ANTHROPIC_API_KEY（platform.claude.com → API Keys；还没有就直接按 Enter）：" value
    echo
    if [ -n "$value" ]; then
        set_env ANTHROPIC_API_KEY "$value"
    else
        warn "还没有 Anthropic 密钥：录音和播放能用，但回答会报错。以后填进 $ENV_FILE 就行。"
    fi
fi
if [ -z "$(get_env CLAUDE_MODEL)" ]; then
    set_env CLAUDE_MODEL claude-sonnet-5
fi
set_env MIC_DEVICE "$MIC_DEVICE"
set_env SPEAKER_DEVICE "$SPEAKER_DEVICE"

step "6/6 后端和按钮开机自动启动"
sudo tee /etc/systemd/system/maxibox.service > /dev/null <<EOF
[Unit]
Description=MaxiBox Backend
After=network-online.target
Wants=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$REPO_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --quiet maxibox
sudo systemctl restart maxibox

started=0
for _ in $(seq 1 30); do
    if curl -fs -o /dev/null http://localhost:8000/; then
        started=1
        break
    fi
    sleep 1
done
if [ "$started" != 1 ]; then
    warn "后端没有启动起来，最近的日志："
    sudo journalctl -u maxibox -n 30 --no-pager || true
    exit 1
fi

# Knopf (GPIO17 an Pin 11, GND an Pin 9): gedrückt halten, sprechen, loslassen
sudo tee /etc/systemd/system/maxibox-box.service > /dev/null <<EOF
[Unit]
Description=MaxiBox Knopf, Aufnahme und Wiedergabe
After=maxibox.service sound.target
Wants=maxibox.service

[Service]
User=$RUN_USER
WorkingDirectory=$REPO_DIR
EnvironmentFile=$ENV_FILE
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV/bin/python $REPO_DIR/pi/box.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --quiet maxibox-box
sudo systemctl restart maxibox-box

ip="$(hostname -I | awk '{print $1}')"
step "完成！"
echo "现在就能用：按住按钮，听到“嘀”一声后说话，说完松开，等回答。"
echo "家长页面：      http://$ip:8000 （同一个 Wi-Fi 下）"
echo "改密钥或模型：  nano $ENV_FILE ，然后 sudo systemctl restart maxibox"
echo "看按钮日志：    journalctl -u maxibox-box -f"
echo "看后端日志：    journalctl -u maxibox -f"
echo "没有按钮时测试：sudo systemctl stop maxibox-box ，然后 $VENV/bin/python $REPO_DIR/pi/talk.py"
