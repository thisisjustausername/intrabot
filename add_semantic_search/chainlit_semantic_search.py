'''
'''

import chainlit as cl

from add_semantic_search.raw_semantic_search import search_urls


@cl.on_message
async def main(message: cl.Message):
    urls = search_urls(query=message.content, k=15, similarity_threshold=0.1)
    if not urls:
        await cl.Message(
            content="No relevant URLs found for your query."
        ).send()
        return

    response_content = "Here are some relevant URLs I found:\n\n"
    for url, prob in urls:
        response_content += f"- ({round(prob * 100, 2)}%) {url}\n"

    await cl.Message(
        content=response_content
    ).send()
