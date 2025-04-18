import asyncio
import numpy as np
import pyaudio
from azure.core.credentials import AzureKeyCredential
from rtclient import (
    InputAudioTranscription,
    NoTurnDetection,
    RTClient,
    RTMessageItem,
    RTResponse,
)

# Configuración de Azure OpenAI
base_url = "wss://tevia01s-swdctrlpc-tcrmopenai-001.openai.azure.com"  # Solo el dominio
key = "54f9e84eef59409cbf1c02ac79c04426"

# Configuración de la grabación
SAMPLE_RATE = 24000
CHUNK_SIZE = int(SAMPLE_RATE * 0.1)  # 100 ms de audio en cada chunk
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Función para enviar audio en tiempo real
async def send_audio(client: RTClient):
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    try:
        print("Grabando y enviando audio en tiempo real...")
        while True:
            audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            await client.send_audio(audio_chunk)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

# Función para recibir la transcripción en tiempo real
async def receive_response(client: RTClient, response: RTResponse):
    async for item in response:
        if item.type == "message":
            async for contentPart in item:
                if contentPart.type == "text":
                    text_data = ""
                    async for chunk in contentPart.text_chunks():
                        text_data += chunk
                    print(f"Transcripción: {text_data}")
        else:
            print("No se ha recibido ningún mensaje de texto.")
    await client.close()

# Ejecutar la transcripción en tiempo real
async def run():
    async with RTClient(
        url=base_url,
        key_credential=AzureKeyCredential(key),
        azure_deployment="gpt-4o-realtime-preview"
    ) as client:
        await client.configure(
            instructions="Transcribe el audio en tiempo real.",
            turn_detection=NoTurnDetection(),
            input_audio_transcription=InputAudioTranscription(model="whisper-1"),
        )

        # Comenzar la captura y transcripción de audio
        response = await client.generate_response()
        send_audio_task = asyncio.create_task(send_audio(client))
        await asyncio.gather(receive_response(client, response), send_audio_task)

# Ejecutar la función principal
if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")
