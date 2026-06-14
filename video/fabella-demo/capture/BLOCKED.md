# Capture Failed

Capture failed: Failed to launch the browser process:  Code: 1

stderr:
Command '/usr/bin/chromium-browser' requires the chromium snap to be installed.
Please install it with:
snap install chromium

TROUBLESHOOTING: https://pptr.dev/troubleshooting


URL: https://build-small-hackathon-fabella.hf.space

## What to try

- Re-run with a longer timeout: `--timeout 60000`
- The site may block headless browsers (anti-bot protection)
- Try capturing a different page on the same domain
