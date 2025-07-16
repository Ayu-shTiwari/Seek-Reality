import { Link } from 'react-router-dom';

const Home = () => (
  <div className="min-h-[50vh]">
    {/* Hero Section */}
    <div className="bg-gradient-to-r from-gray-900 via-indigo-900 to-purple-900 text-white py-20">
      <div className="container mx-auto px-4 text-center">
        <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-400">
          Deepfake Detection System
        </h1>
        <p className="text-xl mb-8 max-w-xl mx-auto text-gray-300">
          Advanced AI-powered solution to detect manipulated videos and images
          Protect yourself from digital deception
        </p>
        <Link
          to="/detect"
          className="px-8 py-4 bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-lg shadow-lg hover:from-cyan-700 hover:to-blue-700 transition-all duration-300 text-lg font-semibold transform hover:scale-105"
        >
          Detect Deepfakes
        </Link>
      </div>
    </div>

    {/* Features Section */}
    <div className="container max-w-screen px-8 py-16 bg-gray-900">
      <h2 className="text-3xl font-bold text-center mb-12 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-400">
        Key Features
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 hover:shadow-xl transition-shadow duration-300">
          <h3 className="text-xl font-semibold mb-4 text-cyan-400">Advanced AI Analysis</h3>
          <p className="text-gray-300">Utilizing state-of-the-art machine learning models to detect subtle manipulation patterns in videos and images.</p>
        </div>
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 hover:shadow-xl transition-shadow duration-300">
          <h3 className="text-xl font-semibold mb-4 text-blue-400">Real-time Detection</h3>
          <p className="text-gray-300">Quick and efficient analysis of uploaded content with detailed results and confidence scores.</p>
        </div>
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 hover:shadow-xl transition-shadow duration-300">
          <h3 className="text-xl font-semibold mb-4 text-purple-400">User-Friendly Interface</h3>
          <p className="text-gray-300">Simple and intuitive platform that makes deepfake detection accessible to everyone.</p>
        </div>
      </div>
    </div>
  </div>
);

export default Home;
  