# #!/bin/bash
# echo "========================================"
# echo "  EmoSense - Emotion Detection App"
# echo "========================================"
# echo ""
# echo "[1/2] Starting Flask backend..."
# python3 app.py &
# FLASK_PID=$!
# sleep 2
# echo "[2/2] Opening frontend in browser..."
# if command -v xdg-open &>/dev/null; then
#     xdg-open index.html
# elif command -v open &>/dev/null; then
#     open index.html
# fi
# echo ""
# echo "Done! API running at http://localhost:5000"
# echo "Press Ctrl+C to stop the server."
# wait $FLASK_PID
#!/bin/bash
echo "========================================"
echo "  EmoSense - Emotion Detection App"
echo "========================================"
echo ""
echo "[1/2] Starting Flask backend..."
python3 app.py &
FLASK_PID=$!
sleep 2
echo "[2/2] Opening frontend in browser..."
if command -v xdg-open &>/dev/null; then
    xdg-open index.html
elif command -v open &>/dev/null; then
    open index.html
fi
echo ""
echo "Done! API running at http://localhost:5000"
echo "Press Ctrl+C to stop the server."
wait $FLASK_PID
