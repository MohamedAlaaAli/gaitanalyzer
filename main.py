import streamlit as st
from gait_analysis import GaitAnalysis
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import StreamlitCallbackHandler
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import uuid
from PIL import Image
import pandas as pd


class GaitQA:
    def __init__(self, pdf_folder="books", persist_dir="chroma_db"):
        self.pdf_folder = pdf_folder
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(model="llama2")
        self.vectorstore = None
        self.llm = Ollama(model="llama2")
        print("initialized QA")

    def build_vector_db(self):
        documents = []
        for file in os.listdir(self.pdf_folder):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.pdf_folder, file))
                documents.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(docs, self.embeddings, persist_directory=self.persist_dir)
        self.vectorstore.persist()
        print("vector DB built")

    def load_vector_db(self):
        print("db loaded")
        self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)

    def ask(self, df: pd.DataFrame, result: str, question: str, return_context_only=False) -> str:
        retriever = self.vectorstore.as_retriever()
        full_query = (
            f"This is the gait data: {df.to_string(index=False)}.\n"
            f"Here is the explanation: {result}\n"
            f"Now, {question}"
        )
        print("asking")
        docs = retriever.get_relevant_documents(full_query)
        context = "\n".join(doc.page_content for doc in docs)

        if return_context_only:
            return context

        full_prompt = f"""Use the following medical documents to answer the question:

{context}

Question: {question}
Answer:"""
        return self.llm(full_prompt)


class StreamlitApp:
    def __init__(self):
        st.set_page_config(
            page_title="Gait Analyzer",
            page_icon="./images/logo.png",
            initial_sidebar_state="collapsed"
        )
        st.title("Gait Analyzer")
        st.sidebar.title("Gait Analyzer")
        st.sidebar.markdown("Source: [GitHub](https://github.com/abishekmuthian/gaitanalyzer).")
        st.sidebar.write("Built by Abishek Muthian. © 2023")
        st.sidebar.markdown("---")

        image = Image.open("./images/logo.png")
        st.caption("Analyze your gait for health disorders at the comfort of your home.")
        st.image(image)
        st.header("Video Upload")

        uploaded_file = st.file_uploader(
            "Choose a short video of you moving from left to right (or) right to left covering your entire body.",
            type="mp4"
        )

        if uploaded_file is not None:
            input_directory = "input_videos"
            os.makedirs(input_directory, exist_ok=True)
            input_video_filename = uuid.uuid4().hex + ".mp4"
            input_path = os.path.join(input_directory, input_video_filename)

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            gait_analysis = GaitAnalysis(input_path)
            output_video, df, result, plt = gait_analysis.process_video()

            st.header("Annotated video:")
            with open(output_video, 'rb') as video_file:
                st.video(video_file.read(), format="video/webm", start_time=0)

            st.header("Plotting the Distances, Peaks and Minima")
            st.subheader("Left Leg: ")
            st.pyplot(plt.figure(1), clear_figure=True)
            st.subheader("Right Leg: ")
            st.pyplot(plt.figure(2), clear_figure=True)

            st.header("Gait Data:")
            st.dataframe(df)

            csv = self.convert_df(df)
            st.download_button("Download Gait Data", csv, "gait_analysis.csv", "text/csv", key='download-csv')

            st.header("Your gait pattern explanation:")

            prompt = (
                "This is my gait data, explain the contents of this gait data and explain my gait pattern from the given gait data - "
                + result
            )

            gait_qa = GaitQA(pdf_folder="books")
            if not os.path.exists("chroma_db"):
                gait_qa.build_vector_db()
            gait_qa.load_vector_db()
            retrieved_context = gait_qa.ask(df, result, prompt, return_context_only=True)

            self.run_model(prompt, context=retrieved_context)

    @staticmethod
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    @staticmethod
    def run_model(prompt, context=None):
        output_container = st.empty().container()
        answer_container = output_container.chat_message("assistant", avatar="🤖")
        st_callback = StreamlitCallbackHandler(answer_container)
        llm = Ollama(model="llama2", callback_manager=CallbackManager([st_callback]))

        full_prompt = prompt
        if context:
            full_prompt = f"""Use the following medical context to improve your analysis of the gait data:

{context}

Now, {prompt}"""

        try:
            llm(full_prompt)
        except Exception as e:
            st.error(f"Cannot access Ollama service: {e}")


if __name__ == "__main__":
    app = StreamlitApp()
