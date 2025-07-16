const About = () => (
  <div className="container max-w-screen px-4 py-12 bg-gray-900">
    <div className="max-w-screen mx-8 ">
      <h2 className="text-3xl font-bold text-center mb-8 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-400">
        About Our Deepfake Detection System
      </h2>
      
      <div className="bg-gray-800 rounded-lg shadow-xl p-8 mb-8 border border-gray-700">
        <h3 className="text-xl font-semibold mb-4 text-cyan-400">Our Mission</h3>
        <p className="text-gray-300 mb-6">
          In an era where digital manipulation is becoming increasingly sophisticated, our mission is to provide reliable and accessible tools to detect deepfakes. We aim to combat misinformation and protect digital integrity through advanced AI technology.
        </p>
        
        <h3 className="text-xl font-semibold mb-4 text-blue-400">Technology</h3>
        <p className="text-gray-300 mb-6">
          Our system utilizes cutting-edge machine learning algorithms and computer vision techniques to analyze videos and images for signs of manipulation. We employ multiple detection methods including:
        </p>
        <ul className="list-disc pl-6 mb-6 text-gray-300 space-y-2">
          <li className="flex items-center">
            <span className="w-2 h-2 bg-cyan-400 rounded-full mr-2"></span>
            Facial manipulation detection
          </li>
          <li className="flex items-center">
            <span className="w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
            Audio-visual synchronization analysis
          </li>
          <li className="flex items-center">
            <span className="w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
            Artifact detection in compressed media
          </li>
          <li className="flex items-center">
            <span className="w-2 h-2 bg-pink-400 rounded-full mr-2"></span>
            Temporal consistency checking
          </li>
        </ul>

        <h3 className="text-xl font-semibold mb-4 text-purple-400">Accuracy and Reliability</h3>
        <p className="text-gray-300 mb-6">
          Our detection system has been trained on extensive datasets of both authentic and manipulated content, achieving high accuracy rates in identifying deepfakes. We continuously update our models to keep pace with evolving manipulation techniques.
        </p>
      </div>

      <div className="bg-gray-800 rounded-lg shadow-xl p-8 border border-gray-700">
        <h3 className="text-xl font-semibold mb-4 text-cyan-400">Contact Us</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 rounded-lg bg-gray-900">
            <h4 className="font-semibold mb-2 text-cyan-400">Email</h4>
            <p className="text-gray-300">support@deepfakedetection.com</p>
          </div>
          <div className="p-4 rounded-lg bg-gray-900">
            <h4 className="font-semibold mb-2 text-blue-400">Phone</h4>
            <p className="text-gray-300">+1 (555) 123-4567</p>
          </div>
          <div className="p-4 rounded-lg bg-gray-900">
            <h4 className="font-semibold mb-2 text-purple-400">Address</h4>
            <p className="text-gray-300">
              123 AI Street<br />
              Tech Valley, CA 94025<br />
              United States
            </p>
          </div>
          <div className="p-4 rounded-lg bg-gray-900">
            <h4 className="font-semibold mb-2 text-pink-400">Business Hours</h4>
            <p className="text-gray-300">
              Monday - Friday: 9:00 AM - 6:00 PM<br />
              Saturday: 10:00 AM - 4:00 PM<br />
              Sunday: Closed
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
);

export default About;  