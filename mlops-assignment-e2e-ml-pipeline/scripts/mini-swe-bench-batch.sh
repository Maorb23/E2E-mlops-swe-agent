MSWEA_COST_TRACKING='ignore_errors' mini-extra swebench \
    --subset verified \
    --split test \
    --model nebius/moonshotai/Kimi-K2.6 \
    --slice '0:3' \
    --config mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml \
    --workers 5 \
    -o trajectories
