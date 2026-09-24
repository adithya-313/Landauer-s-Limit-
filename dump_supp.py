import json
d = json.load(open('supplementary_results.json'))

with open('supplementary_raw.txt', 'w') as f:
    f.write('Condition A: Cache Disabled\n')
    for i, x in enumerate(d['cache_disabled_500']):
        f.write(f'Req {i+1}/20 - Loop: {x["possible_loop"]}, Hit: {x["hit"]}, Saved: {x["saved"]}, TTFT: {x["ttft"]:.4f}s\n')
        
    f.write('\nCondition B: Cache Enabled\n')
    for i, x in enumerate(d['cache_enabled_500']):
        f.write(f'Req {i+1}/20 - Loop: {x["possible_loop"]}, Hit: {x["hit"]}, Saved: {x["saved"]}, TTFT: {x["ttft"]:.4f}s\n')
