"""Authenticated public processing API; runs only behind the outbound HTTPS tunnel."""
from mac_server import create_app, ROOT
app=create_app(data_root=ROOT/'runs-public/jobs',public_project='voxelflight-3d')
