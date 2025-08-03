'use client';
import { useRef, useState, useEffect } from 'react';

export default function WebcamCapture() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [emotion, setEmotion] = useState('N/A');
    
    useEffect(() => {
        async function enableCam() {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
        }
        enableCam();
    }, []);
    
    const CaptureAndSend = async () => {
        const canvas = canvasRef.current;
        const video = videoRef.current;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        canvas.toBlob(async (blob) => {
            if (!blob) {
                console.error('Blob is null, skipping submission.');
                setEmotion('Error: No image captured');
                return;
            }
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            try {
                const response = await fetch('http://127.0.0.1:8000/predict', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();
                setEmotion(data.emotion || 'Unknown');
            } catch (err) {
                setEmotion('Error');
                console.error(err);
            }
        }, 'image/jpeg');
    }
    
    return (
        <div className="flex flex-col items-center gap-4">
            <video ref={videoRef} autoPlay className="rounded shadow w-[300px]" />
            <canvas ref={canvasRef} className="hidden" />
            <button onClick={CaptureAndSend} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                Detect Emotion
            </button>
            <p className="text-lg font-medium text-gray-800">Detected Emotion: {emotion}</p>
        </div>
    );
} 