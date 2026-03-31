import gradio as gr
from quran_rag_agent import process_query

last_retrieved_chunks = []

def get_retrieved_chunks():
    """Return formatted retrieved chunks for display."""
    global last_retrieved_chunks
    if not last_retrieved_chunks:
        return "لم يتم استرجاع نتائج بعد (No results retrieved yet)"
    
    output = ""
    for i, chunk in enumerate(last_retrieved_chunks, 1):
        meta = chunk["metadata"]
        output += f"### مصدر {i}\n"
        output += f"**السورة:** {meta.get('surah_name', 'N/A')} | "
        output += f"**نوع الوحي:** {meta.get('revelation_type', 'N/A')} | "
        output += f"**كتاب التفسير:** {meta.get('tafsir_book', 'N/A')}\n\n"
        output += f"{chunk['content']}\n\n---\n\n"
    return output

with gr.Blocks(title="المساعد القرآني الذكي (AI Quran Expert)") as demo:
    gr.Markdown("# AI Quran Expert")
    gr.Markdown("اسأل عن أي آية أو موضوع، وسيقوم النظام بالبحث في القرآن وكتب التفسير.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="المحادثة", height=500)
            msg = gr.Textbox(label="سؤالك", placeholder="اكتب سؤالك هنا...", rtl=True)
            with gr.Accordion("Advanced Settings", open=False):
                k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="k")
            with gr.Row():
                submit_btn = gr.Button("إرسال", variant="primary")
                clear_btn = gr.Button("مسح المحادثة")
        
        with gr.Column(scale=1):
            gr.Markdown("## المصادر")
            with gr.Accordion("عرض المصادر", open=False):
                sources_display = gr.Markdown("لم يتم استرجاع نتائج بعد")
                refresh_btn = gr.Button("🔄 تحديث", size="sm")
    
    gr.Examples(
        examples=["تفسير آية الكرسي", "شرح سورة الإخلاص"],
        inputs=msg
    )
    
    def respond(message, chat_history, k_val):
        global last_retrieved_chunks
        response, chunks = process_query(message, k_val)
        last_retrieved_chunks = chunks
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})
        return "", chat_history, get_retrieved_chunks()
    
    submit_btn.click(respond, [msg, chatbot, k_slider], [msg, chatbot, sources_display])
    msg.submit(respond, [msg, chatbot, k_slider], [msg, chatbot, sources_display])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, sources_display])
    refresh_btn.click(get_retrieved_chunks, outputs=sources_display)

if __name__ == "__main__":
    demo.launch()
