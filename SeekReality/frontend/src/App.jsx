import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './components/Home';
import About from './components/About';
import Navbar from './components/Navbar';
import VideoUpload from './components/VideoUpload';
import Footer from './components/Footer';
import ProtectedRoute from './components/ProtectedRoute';

const App = () => (
  <Router>
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route element={<ProtectedRoute />}>
            <Route path="/detect" element={<VideoUpload />} />
          </Route>
        </Routes>
      </main>
      <Footer />
    </div>
  </Router>
);

export default App;