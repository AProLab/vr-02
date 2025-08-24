import streamlit as st
from openai import OpenAI
import io


class OpenAIClient:
    """OpenAI API 연동 담당 클래스"""
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = None
        if api_key:
            self.set_api_key(api_key)

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def transcribe(self, file):
        if not self.client:
            raise ValueError("API Key가 설정되지 않았습니다.")
        return self.client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=file
        )

    def get_response(self, transcript_text: str):
        if not self.client:
            raise ValueError("API Key가 설정되지 않았습니다.")
        return self.client.responses.create(
            model="gpt-4.1-mini",
            input=transcript_text
        )


class VoiceFileHandler:
    """음성 파일 업로드 및 처리 담당 클래스"""
    def __init__(self):
        self.file_voice = None

    def upload(self):
        self.file_voice = st.file_uploader("음성 파일만 업로드하세요!", type=['mp3', 'wav', 'm4a'])
        if self.file_voice:
            st.audio(self.file_voice)  # 미리 듣기
        return self.file_voice

    def get_bytesio(self):
        if self.file_voice:
            audio_bytes = self.file_voice.read()
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = self.file_voice.name
            return audio_file
        return None


class TextResponseApp:
    """앱 실행 및 UI 관리"""
    def __init__(self):
        self.client = OpenAIClient()
        self.file_handler = VoiceFileHandler()
        self.transcript = None

    def input_api_key(self):
        api_key = st.text_input("OPENAI API KEY를 입력후, 엔터키를 입력하세요.", type="password")
        if api_key:
            self.client.set_api_key(api_key)

    def transcribe_audio(self):
        audio_file = self.file_handler.get_bytesio()
        if audio_file:
            with st.spinner("음성을 텍스트로 변환중..."):
                result = self.client.transcribe(audio_file)
                self.transcript = result.text

    def display_result(self):
        if self.transcript:
            with st.spinner("답변 생성중..."):
                response = self.client.get_response(self.transcript)
            st.subheader("답변")
            st.markdown(response.output_text)

    def run(self):
        st.header("음성 질문 → 문자 답변")
        self.input_api_key()
        if self.client.api_key:
            self.file_handler.upload()
            if self.file_handler.file_voice:
                self.transcribe_audio()
                self.display_result()


if __name__ == "__main__":
    app = TextResponseApp()
    app.run()