import { Link, useLocation } from 'react-router-dom';
import { useAuth0 } from "@auth0/auth0-react";
import React from "react";

const Navbar = () => {
  const location = useLocation();
  const { loginWithRedirect, logout, user, isAuthenticated } = useAuth0();

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="bg-gradient-to-r from-gray-900 to-indigo-900 shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center">
            <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-400">
              Deepfake Detection
            </span>
          </Link>

          {/* User Info & Auth Buttons */}
          <div className="flex items-center space-x-4">
            {isAuthenticated && user && (
              <div className="flex items-center space-x-2">
                <img src={user.picture} alt={user.nickname} className="w-8 h-8 rounded-full border-2 border-cyan-400" />
                <span className="text-sm font-medium text-gray-300">Welcome, {user.nickname}!</span>
              </div>
            )}
            {isAuthenticated ? (
              <div className="hidden md:flex space-x-8">
                <Link
                  to="/"
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-300 ${
                    isActive('/')
                      ? 'text-cyan-400 bg-gray-800'
                      : 'text-gray-300 hover:text-cyan-400 hover:bg-gray-800'
                  }`}
                >
                  Home
                </Link>
                <Link
                  to="/detect"
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-300 ${
                    isActive('/detect')
                      ? 'text-cyan-400 bg-gray-800'
                      : 'text-gray-300 hover:text-cyan-400 hover:bg-gray-800'
                  }`}
                >
                  Detect
                </Link>
                <Link
                  to="/about"
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-300 ${
                    isActive('/about')
                      ? 'text-cyan-400 bg-gray-800'
                      : 'text-gray-300 hover:text-cyan-400 hover:bg-gray-800'
                  }`}
                >
                  About
                </Link>
                <button
                  onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
                  className="px-4 py-2 bg-gradient-to-r from-red-600 to-pink-600 text-white rounded-md text-sm font-medium hover:from-red-700 hover:to-pink-700 transition-all duration-300 transform hover:scale-105"
                >
                  Log Out
                </button>
              </div>
            ) : (
              <button
                onClick={() => loginWithRedirect()}
                className="px-4 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-md text-sm font-medium hover:from-cyan-700 hover:to-blue-700 transition-all duration-300 transform hover:scale-105"
              >
                Log In
              </button>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;