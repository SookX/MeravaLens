import { BrowserRouter, Route, Routes } from "react-router-dom"
import { Box, ThemeProvider } from "@mui/material"
import { theme } from "./theme/theme"
import DataProvider from "./context/DataContext"
import { lazyLoad } from "./lazyLoad"

// import Home from "./pages/Home/Home"
// import Register from "./pages/Register/Register"
// import Login from "./pages/Login/Login"
// import Activate from "./pages/Activate/Activate"
// import ResetPass from "./pages/ResetPass/ResetPass"
// import ForgotPass from "./components/ForgotPass/ForgotPass"
// import Dashboard from "./pages/Dashboard/Dashboard"
// import ChangePassword from "./pages/ChangePassword/ChangePassword"

const Home = lazyLoad("/src/pages/Home/Home")
const Register = lazyLoad("/src/pages/Register/Register")
const Login = lazyLoad("/src/pages/Login/Login")
const Activate = lazyLoad("/src/pages/Activate/Activate")
const ResetPass = lazyLoad("/src/pages/ResetPass/ResetPass")
const ForgotPass = lazyLoad("/src/components/ForgotPass/ForgotPass")
const Dashboard = lazyLoad("/src/pages/Dashboard/Dashboard")
const ChangePassword = lazyLoad("/src/pages/ChangePassword/ChangePassword")

import { GoogleOAuthProvider } from "@react-oauth/google"
import RouteWrapper from "./components/RouteWrapper/RouteWrapper"

function App() {
  const clientId = import.meta.env.VITE_GOOGLE_OAUTH2

  return (
    <ThemeProvider theme={theme}>
      <Box bgcolor="background.default" sx={{ minWidth: "100vw", minHeight: "100vh" }}>
        <BrowserRouter>
          <DataProvider>
            <GoogleOAuthProvider clientId={clientId}>
              <Routes>
                <Route path="/" element={<RouteWrapper />}>
                  <Route index element={<Home />} />
                  <Route path="register" element={<Register />} />
                  <Route path="login" element={<Login />} />
                  <Route path="activate/:uidb64/:token" element={<Activate />} />
                  <Route path="forgot-password" element={<ForgotPass />} />
                  <Route path="reset-password/:uidb64/:token" element={<ResetPass />} />
                  <Route path="change-password" element={<ChangePassword />} />
                  <Route path="dashboard" element={<Dashboard />} />
                </Route>
              </Routes>
            </GoogleOAuthProvider>
          </DataProvider>
        </BrowserRouter>
      </Box>
    </ThemeProvider>
  )
}

export default App
