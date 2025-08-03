import WebcamCapture from '../components/WebcamCapture';

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-gray-100 py-10">
      <h1 className="text-3xl font-bold mb-6">Webcam Emotion Detection</h1>
      <WebcamCapture />
    </main>
  );
}
