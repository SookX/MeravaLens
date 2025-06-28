import { BrowserRouter, Route, Routes } from "react-router-dom"
import Home from "./pages/Home/Home"
import { Box, ThemeProvider } from "@mui/material"
import { theme } from "./theme/theme"
import DataProvider from "./context/DataContext"
import Register from "./pages/Register/Register"
import Login from "./pages/Login/Login"
import Activate from "./pages/Activate/Activate"

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Box bgcolor="background.default" sx={{ minWidth: "100vw", minHeight: "100vh" }}>
        <BrowserRouter>
          <DataProvider>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/register" element={<Register />} />
              <Route path="/login" element={<Login />} />
              <Route path="/activate/:uidb64/:token" element={<Activate />} />
            </Routes>
          </DataProvider>
        </BrowserRouter>
      </Box>
    </ThemeProvider>
  )
}

export default App
