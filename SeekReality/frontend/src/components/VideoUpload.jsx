import { useState, useRef } from 'react';
import { useAuth0 } from "@auth0/auth0-react";
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

// New component for video results
const VideoResults = ({ result }) => {
  const reportRef = useRef(null);

  const downloadPDF = async () => {
    if (!reportRef.current) return;

    // Create a clone of the content for PDF
    const contentClone = reportRef.current.cloneNode(true);
    
    // Create a new container for the PDF content
    const pdfContainer = document.createElement('div');
    pdfContainer.style.backgroundColor = '#1f2937';
    pdfContainer.style.padding = '20px';
    pdfContainer.style.color = '#e5e7eb';
    pdfContainer.style.fontFamily = 'Arial, sans-serif';

    // Create the main result section
    const mainResult = document.createElement('div');
    mainResult.style.backgroundColor = '#374151';
    mainResult.style.padding = '20px';
    mainResult.style.marginBottom = '20px';
    mainResult.style.borderRadius = '8px';

    const mainTitle = document.createElement('h3');
    mainTitle.textContent = 'Overall Analysis';
    mainTitle.style.color = '#22d3ee';
    mainTitle.style.fontSize = '20px';
    mainTitle.style.marginBottom = '15px';
    mainResult.appendChild(mainTitle);

    const statsGrid = document.createElement('div');
    statsGrid.style.display = 'grid';
    statsGrid.style.gridTemplateColumns = '1fr 1fr';
    statsGrid.style.gap = '20px';

    // Left column
    const leftCol = document.createElement('div');
    leftCol.innerHTML = `
      <p><strong>Status:</strong> <span style="color: ${result.isDeepfake ? '#ef4444' : '#34d399'}">${result.isDeepfake ? 'Likely Deepfake' : 'Likely Authentic'}</span></p>
      <p><strong>Confidence:</strong> ${Math.round(result.confidence * 100)}%</p>
      <p><strong>Temporal Consistency:</strong> ${Math.round(result.consistency * 100)}%</p>
    `;
    statsGrid.appendChild(leftCol);

    // Right column
    const rightCol = document.createElement('div');
    rightCol.innerHTML = `
      <p><strong>Average Prediction:</strong> ${Math.round(result.statistics.average * 100)}%</p>
      <p><strong>Median Prediction:</strong> ${Math.round(result.statistics.median * 100)}%</p>
      <p><strong>Lower Quartile:</strong> ${Math.round(result.statistics.lower_quartile * 100)}%</p>
    `;
    statsGrid.appendChild(rightCol);

    mainResult.appendChild(statsGrid);
    pdfContainer.appendChild(mainResult);

    // Add key frames if they exist
    if (result.frame_results && result.frame_results.length > 0) {
      const framesSection = document.createElement('div');
      framesSection.style.backgroundColor = '#374151';
      framesSection.style.padding = '20px';
      framesSection.style.marginBottom = '20px';
      framesSection.style.borderRadius = '8px';

      const framesTitle = document.createElement('h3');
      framesTitle.textContent = 'Key Frames Analysis';
      framesTitle.style.color = '#22d3ee';
      framesTitle.style.fontSize = '20px';
      framesTitle.style.marginBottom = '15px';
      framesSection.appendChild(framesTitle);

      const framesGrid = document.createElement('div');
      framesGrid.style.display = 'grid';
      framesGrid.style.gridTemplateColumns = 'repeat(3, 1fr)';
      framesGrid.style.gap = '15px';

      result.frame_results.forEach((frame, index) => {
        const frameContainer = document.createElement('div');
        frameContainer.style.position = 'relative';
        frameContainer.style.marginBottom = '15px';

        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${frame.image}`;
        img.style.width = '100%';
        img.style.borderRadius = '8px';
        frameContainer.appendChild(img);

        const frameInfo = document.createElement('div');
        frameInfo.style.position = 'absolute';
        frameInfo.style.bottom = '0';
        frameInfo.style.left = '0';
        frameInfo.style.right = '0';
        frameInfo.style.backgroundColor = 'rgba(0, 0, 0, 0.75)';
        frameInfo.style.padding = '8px';
        frameInfo.style.borderBottomLeftRadius = '8px';
        frameInfo.style.borderBottomRightRadius = '8px';
        frameInfo.innerHTML = `
          <span style="color: ${frame.result === 'Real' ? '#34d399' : '#ef4444'}">${frame.result}</span>
          (${Math.round(frame.prediction * 100)}%)
        `;
        frameContainer.appendChild(frameInfo);
        framesGrid.appendChild(frameContainer);
      });

      framesSection.appendChild(framesGrid);
      pdfContainer.appendChild(framesSection);
    }

    // Add prediction timeline if it exists
    if (result.frame_predictions && result.frame_predictions.length > 0) {
      const timelineSection = document.createElement('div');
      timelineSection.style.backgroundColor = '#374151';
      timelineSection.style.padding = '20px';
      timelineSection.style.marginBottom = '20px';
      timelineSection.style.borderRadius = '8px';

      const timelineTitle = document.createElement('h3');
      timelineTitle.textContent = 'Prediction Timeline';
      timelineTitle.style.color = '#22d3ee';
      timelineTitle.style.fontSize = '20px';
      timelineTitle.style.marginBottom = '15px';
      timelineSection.appendChild(timelineTitle);

      const timelineContainer = document.createElement('div');
      timelineContainer.style.height = '100px';
      timelineContainer.style.position = 'relative';
      timelineContainer.style.marginBottom = '10px';

      // Create the base line
      const baseLine = document.createElement('div');
      baseLine.style.position = 'absolute';
      baseLine.style.top = '50%';
      baseLine.style.left = '0';
      baseLine.style.right = '0';
      baseLine.style.height = '2px';
      baseLine.style.backgroundColor = '#4b5563';
      timelineContainer.appendChild(baseLine);

      // Add prediction bars
      result.frame_predictions.forEach((prediction, index) => {
        const bar = document.createElement('div');
        bar.style.position = 'absolute';
        bar.style.bottom = '0';
        bar.style.width = '2px';
        bar.style.height = `${prediction * 100}%`;
        bar.style.backgroundColor = prediction > 0.6 ? '#34d399' : '#ef4444';
        bar.style.left = `${(index / (result.frame_predictions.length - 1)) * 100}%`;
        timelineContainer.appendChild(bar);
      });

      timelineSection.appendChild(timelineContainer);

      // Add start/end labels
      const labelsContainer = document.createElement('div');
      labelsContainer.style.display = 'flex';
      labelsContainer.style.justifyContent = 'space-between';
      labelsContainer.style.color = '#9ca3af';
      labelsContainer.style.fontSize = '14px';
      labelsContainer.innerHTML = '<span>Start</span><span>End</span>';
      timelineSection.appendChild(labelsContainer);

      pdfContainer.appendChild(timelineSection);
    }

    // Create a temporary container for PDF generation
    const tempContainer = document.createElement('div');
    tempContainer.style.position = 'absolute';
    tempContainer.style.left = '-9999px';
    tempContainer.style.top = '-9999px';
    tempContainer.appendChild(pdfContainer);
    document.body.appendChild(tempContainer);

    try {
      const canvas = await html2canvas(tempContainer, {
        backgroundColor: '#1f2937',
        scale: 2,
        logging: false,
        useCORS: true,
        allowTaint: true
      });
      
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save('deepfake-analysis-report.pdf');
    } finally {
      document.body.removeChild(tempContainer);
    }
  };

  return (
    <div className="space-y-6">
      <div ref={reportRef} className="space-y-6">
        {/* Main Result */}
        <div className="p-4 bg-gradient-to-r from-[#1a1a1a] to-[#1e3a8a] rounded-lg border border-gray-700">
          <h3 className="text-xl font-semibold mb-4 text-[#22d3ee]">Overall Analysis</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-gray-300">
                <span className="font-semibold">Status:</span>{' '}
                <span className={result.isDeepfake ? 'text-red-400' : 'text-green-400'}>
                  {result.isDeepfake ? 'Likely Deepfake' : 'Likely Authentic'}
                </span>
              </p>
              <p className="text-gray-300">
                <span className="font-semibold">Confidence:</span>{' '}
                {Math.round(result.confidence * 100)}%
              </p>
              <p className="text-gray-300">
                <span className="font-semibold">Temporal Consistency:</span>{' '}
                {Math.round(result.consistency * 100)}%
              </p>
            </div>
            <div>
              <p className="text-gray-300">
                <span className="font-semibold">Average Prediction:</span>{' '}
                {Math.round(result.statistics.average * 100)}%
              </p>
              <p className="text-gray-300">
                <span className="font-semibold">Median Prediction:</span>{' '}
                {Math.round(result.statistics.median * 100)}%
              </p>
              <p className="text-gray-300">
                <span className="font-semibold">Lower Quartile:</span>{' '}
                {Math.round(result.statistics.lower_quartile * 100)}%
              </p>
            </div>
          </div>
        </div>

        {/* Key Frames */}
        {result.frame_results && result.frame_results.length > 0 && (
          <div className="p-4 bg-gradient-to-r from-[#1a1a1a] to-[#1e3a8a] rounded-lg border border-gray-700">
            <h3 className="text-xl font-semibold mb-4 text-[#22d3ee]">Key Frames Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {result.frame_results.map((frame, index) => (
                <div key={index} className="relative">
                  <img
                    src={`data:image/jpeg;base64,${frame.image}`}
                    alt={`Frame ${index + 1}`}
                    className="w-full h-auto rounded-lg shadow-lg"
                  />
                  <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 p-2 rounded-b-lg">
                    <p className="text-sm text-white">
                      <span className={frame.result === 'Real' ? 'text-green-400' : 'text-red-400'}>
                        {frame.result}
                      </span>
                      {' '}({Math.round(frame.prediction * 100)}%)
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Prediction Timeline */}
        {result.frame_predictions && result.frame_predictions.length > 0 && (
          <div className="p-4 bg-gradient-to-r from-[#1a1a1a] to-[#1e3a8a] rounded-lg border border-gray-700">
            <h3 className="text-xl font-semibold mb-4 text-[#22d3ee]">Prediction Timeline</h3>
            <div className="h-32 relative">
              {/* Base line */}
              <div className="absolute inset-0 flex items-center">
                <div className="w-full h-1 bg-gray-700"></div>
              </div>
              {/* Prediction bars */}
              {result.frame_predictions.map((prediction, index) => (
                <div
                  key={index}
                  className="absolute bottom-0 w-1 h-8"
                  style={{
                    left: `${(index / (result.frame_predictions.length - 1)) * 100}%`,
                    backgroundColor: prediction > 0.6 ? '#34d399' : '#ef4444',
                    transform: `scaleY(${prediction})`,
                    transformOrigin: 'bottom'
                  }}
                />
              ))}
            </div>
            <div className="flex justify-between mt-2 text-sm text-gray-400">
              <span>Start</span>
              <span>End</span>
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-center">
        <button
          onClick={downloadPDF}
          className="px-6 py-3 bg-[#059669] text-white rounded-lg hover:bg-[#047857] transition-all duration-300 transform hover:scale-105"
        >
          Download PDF Report
        </button>
      </div>
    </div>
  );
};

// Main component
const VideoUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);
  const { user, isAuthenticated } = useAuth0();

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith('video/') || file.type.startsWith('image/')) {
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setError(null);
        setResult(null);
      } else {
        setError('Please upload a video or image file');
        setSelectedFile(null);
        setPreviewUrl(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const endpoint = selectedFile.type.startsWith('video/') 
        ? 'http://localhost:5000/api/analyze-video'
        : 'http://localhost:5000/api/analyze-image';

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        if (selectedFile.type.startsWith('video/')) {
          setResult({
            isDeepfake: data.result === 'Fake',
            confidence: data.confidence,
            consistency: data.consistency,
            statistics: data.statistics,
            frame_results: data.frame_results,
            frame_predictions: data.frame_predictions
          });
        } else {
          setResult({
            isDeepfake: data.result === 'Fake',
            confidence: data.confidence,
            image: data.image
          });
        }
      } else {
        setError(data.error || 'Detection failed');
      }
    } catch (err) {
      setError('An error occurred during detection');
      console.error('Upload error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="container max-w-screen mx-auto px-4 py-8 bg-gray-900">
      <div className="max-w-4xl mx-auto bg-gray-900">
        <div className="bg-[#1f2937] rounded-lg shadow-xl p-6 border border-gray-700">
          <h2 className="text-3xl font-bold mb-6 text-center bg-clip-text text-transparent bg-gradient-to-r from-[#22d3ee] to-[#3b82f6]">
            Deepfake Detection
          </h2>

          <div className="mb-8">
            <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center bg-gray-900">
              {previewUrl ? (
                <div className="relative">
                  {selectedFile.type.startsWith('video/') ? (
                    <video
                      src={previewUrl}
                      controls
                      className="max-w-full h-auto mx-auto rounded-lg shadow-lg"
                    />
                  ) : (
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="max-w-full h-auto mx-auto rounded-lg shadow-lg"
                    />
                  )}
                  <button
                    onClick={handleReset}
                    className="absolute top-2 right-2 bg-gradient-to-r from-red-600 to-pink-600 text-white p-2 rounded-full hover:from-red-700 hover:to-pink-700 transition-all duration-300 transform hover:scale-110"
                  >
                    ×
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  <p className="text-gray-300">Upload a video or image to check for deepfake manipulation</p>
                  <input
                    type="file"
                    accept="video/*,image/*"
                    onChange={handleFileChange}
                    ref={fileInputRef}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="px-6 py-3 bg-gradient-to-r from-[#0891b2] to-[#2563eb] text-white rounded-lg cursor-pointer hover:from-[#0e7490] hover:to-[#1d4ed8] transition-all duration-300 transform hover:scale-105"
                  >
                    Choose File
                  </label>
                </div>
              )}
            </div>
          </div>

          {error && (
            <div className="mb-4 p-4 bg-gradient-to-r from-[#7f1d1d]/50 to-[#be185d]/50 text-red-300 rounded-lg border border-red-700">
              {error}
            </div>
          )}

          {result && (
            <div className="mb-4">
              {selectedFile.type.startsWith('video/') ? (
                <VideoResults result={result} />
              ) : (
                <div className="p-4 bg-gradient-to-r from-[#1a1a1a] to-[#1e3a8a] rounded-lg border border-gray-700">
                  <div ref={reportRef}>
                    <h3 className="text-xl font-semibold mb-2 text-[#22d3ee]">Detection Results</h3>
                    <div className="space-y-2">
                      <p>
                        <span className="font-semibold text-gray-300">Status:</span>{' '}
                        <span className={result.isDeepfake ? 'text-red-400' : 'text-green-400'}>
                          {result.isDeepfake ? 'Likely Deepfake' : 'Likely Authentic'}
                        </span>
                      </p>
                      <p>
                        <span className="font-semibold text-gray-300">Confidence:</span>{' '}
                        {Math.round(result.confidence * 100)}%
                      </p>
                      {result.image && (
                        <div className="mt-4">
                          <p className="font-semibold text-gray-300 mb-2">Processed Image:</p>
                          <img
                            src={`data:image/jpeg;base64,${result.image}`}
                            alt="Processed"
                            className="max-w-full h-auto rounded-lg shadow-lg"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex justify-center mt-4">
                    <button
                      onClick={downloadPDF}
                      className="px-6 py-3 bg-[#059669] text-white rounded-lg hover:bg-[#047857] transition-all duration-300 transform hover:scale-105"
                    >
                      Download PDF Report
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="flex justify-center space-x-4">
            <button
              onClick={handleUpload}
              disabled={!selectedFile || loading}
              className={`px-6 py-3 rounded-lg text-white font-semibold transition-all duration-300 transform hover:scale-105 ${
                !selectedFile || loading
                  ? 'bg-[#4b5563] cursor-not-allowed'
                  : 'bg-gradient-to-r from-[#0891b2] to-[#2563eb] hover:from-[#0e7490] hover:to-[#1d4ed8]'
              }`}
            >
              {loading ? 'Analyzing...' : 'Detect Deepfake'}
            </button>
            {selectedFile && (
              <button
                onClick={handleReset}
                className="px-6 py-3 bg-gradient-to-r from-[#4b5563] to-[#374151] text-white rounded-lg hover:from-[#374151] hover:to-[#1f2937] transition-all duration-300 transform hover:scale-105"
              >
                Reset
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoUpload; 