The procedure for using Discord whenever there's a push from the github to the hugging face space:
- Creating a new server
- Server Settings -> Integrations -> Create Webhook
- Copy Webhook URL
- Go to Github Repository Settings
- Choose Webhooks
- Paste URL to Payload URL
- Add /github to the end of URL
- Click "Just the Push Events"
- Add Webhook

Example: <img width="1265" alt="Screenshot 1403-06-25 at 23 25 01" src="https://github.com/user-attachments/assets/43d6447f-006e-4112-9668-532fa5cd4f75">




---
title: DIS Background Removal
emoji: 🔥 🌠 🏰
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 4.41.0
python_version: 3.11.9
app_file: app.py
pinned: false
license: apache-2.0
models:
- doevent/dis
---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference



