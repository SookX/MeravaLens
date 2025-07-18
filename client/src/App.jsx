import { BrowserRouter, Route, Routes } from "react-router-dom"
import Home from "./pages/Home/Home"
import { Box, ThemeProvider } from "@mui/material"
import { theme } from "./theme/theme"
import DataProvider from "./context/DataContext"
import Register from "./pages/Register/Register"
import Login from "./pages/Login/Login"
import Activate from "./pages/Activate/Activate"
import ResetPass from "./pages/ResetPass/ResetPass"
import ForgotPass from "./components/ForgotPass/ForgotPass"
import { GoogleOAuthProvider } from "@react-oauth/google"
import Dashboard from "./pages/Dashboard/Dashboard"
import Header from "./components/Header/Header"
import ChangePassword from "./pages/ChangePassword/ChangePassword"
import Footer from "./components/Footer/Footer"

function App() {
  const clientId = import.meta.env.VITE_GOOGLE_OAUTH2

  return (
    <ThemeProvider theme={theme}>
      <Box bgcolor="background.default" sx={{ minWidth: "100vw", minHeight: "100vh" }}>
        <BrowserRouter>
          <DataProvider>
            <GoogleOAuthProvider clientId={clientId}>
              <Header />
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/register" element={<Register />} />
                <Route path="/login" element={<Login />} />
                <Route path="/activate/:uidb64/:token" element={<Activate />} />
                <Route path="/forgot-password" element={<ForgotPass />} />
                <Route path="/reset-password/:uidb64/:token" element={<ResetPass />} />
                <Route path="/change-password" element={<ChangePassword />} />
                <Route path="/dashboard" element={<Dashboard />} />
              </Routes>
              <Footer />
            </GoogleOAuthProvider>
          </DataProvider>
        </BrowserRouter>
      </Box>
    </ThemeProvider>
  )
}

export default App
