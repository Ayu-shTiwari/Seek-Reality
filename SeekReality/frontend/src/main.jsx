import { createRoot } from 'react-dom/client'
import { Auth0Provider } from '@auth0/auth0-react';
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <Auth0Provider
    domain="dev-zeageaxww6hbcnq0.us.auth0.com"
    clientId="oS5J1dN0Y4vvH5nkKiCfJwIpeLGJYtSI"
    authorizationParams={{
      redirect_uri: window.location.origin
    }}
  >
    <App />
  </Auth0Provider>,
)
