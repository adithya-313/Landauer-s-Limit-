import asyncio
import json
from semantic_cache import DeterministicSemanticCache

async def main():
    sc = DeterministicSemanticCache()
    
    # 5-6 distinct topics/messages
    messages_list = [
        # Expected HIT group 1 (Docker)
        ([{'role': 'user', 'content': 'What is Docker?'}], 'Docker is a container platform.'),
        ([{'role': 'user', 'content': 'Explain Docker simply.'}], 'Docker is a container platform.'),
        
        # Expected HIT group 2 (Python)
        ([{'role': 'user', 'content': 'How to install Python?'}], 'Use apt-get install python3.'),
        ([{'role': 'user', 'content': 'Python installation guide'}], 'Use apt-get install python3.'),
        
        # Expected MISS group 1 (Unrecognized entity)
        ([{'role': 'user', 'content': 'Tell me a joke.'}], 'Why did the chicken cross the road?'),
        ([{'role': 'user', 'content': 'Tell me another joke.'}], 'Knock knock.'),
        
        # Expected HIT group 3 (Kubernetes)
        ([{'role': 'user', 'content': 'What is Kubernetes?'}], 'K8s is an orchestration tool.'),
        ([{'role': 'user', 'content': 'Explain Kubernetes to me.'}], 'K8s is an orchestration tool.'),
        
        # Expected MISS group 2 (Ambiguous - multiple entities)
        ([{'role': 'user', 'content': 'Should I use Python or Java?'}], 'It depends on your use case.'),
        ([{'role': 'user', 'content': 'Compare Python and Java.'}], 'It depends on your use case.'),
    ]
    
    print('Generating misses and populating cache...')
    for msgs, resp in messages_list:
        await sc.check_cache(msgs) # Miss
        await sc.insert(msgs, {'choices': [{'message': {'content': resp}}]})
    
    print('Generating hits/misses from varied traffic...')
    for _ in range(5):
        for msgs, _ in messages_list:
            await sc.check_cache(msgs)

if __name__ == '__main__':
    asyncio.run(main())
