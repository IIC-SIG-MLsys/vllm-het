# python3 ./vllm_het/start.py

cp -f vllm_het/vllm_auto_patch.py /usr/local/lib/python3.12/dist-packages/
cp -f vllm_het/p2p_backend.py /usr/local/lib/python3.12/dist-packages/

cat >/usr/local/lib/python3.12/dist-packages/vllm_patch.pth <<'EOF'
import vllm_auto_patch  # noqa: F401
EOF
