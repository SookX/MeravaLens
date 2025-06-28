import { BrowserRouter, Route, Routes } from "react-router-dom"
import Home from "./pages/Home/Home"
import { Box, ThemeProvider } from "@mui/material"
import { theme } from "./theme/theme"

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Box bgcolor="background.default" sx={{ minWidth: "100vw", minHeight: "100vh" }}>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Home />} />
          </Routes>
        </BrowserRouter>
      </Box>
    </ThemeProvider>
  )
}

export default App
