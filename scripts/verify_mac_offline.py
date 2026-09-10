"""Actual four-view GPU run with Python TCP connections forbidden, not a mock."""
from pathlib import Path
import runpy
import socket
import sys
from unittest.mock import patch

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.argv=[str(root/'mac_reconstruct.py'),'--images',str(root/'mac-smoke-input'),'--output',str(root/'runs-mac/offline-textured-proof'),'--max-frames','4','--size','350']
def blocked(*args,**kwargs):raise RuntimeError('OFFLINE TEST: outbound network is forbidden')
with patch.object(socket.socket,'connect',blocked),patch.object(socket.socket,'connect_ex',blocked),patch.object(socket,'create_connection',blocked):
    runpy.run_path(str(root/'mac_reconstruct.py'),run_name='__main__')
