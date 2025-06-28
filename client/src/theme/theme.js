import { createTheme } from "@mui/material"

const palette = {
    mode: "dark",
    background: {
        default: "#000"
    },
    text: {
        default: "#fff"
    }
}

export const theme = createTheme({
    palette,
    typography: {
        allVariants: {
            color: palette.text.default
        }
    }
})