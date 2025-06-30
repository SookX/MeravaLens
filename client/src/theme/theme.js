import { createTheme } from "@mui/material"

const palette = {
    mode: "dark",
    background: {
        default: "#000",
        light: "#333333"
    },
    text: {
        default: "#DDE6ED",
        secondary: "#b1b8be",
        dark: "#2c2e2f"
    },
    primary: {
        main: "#6EACDA",
        dark: "#588aae"
    }
}

const headingStyles = {
    fontFamily: ["Aldrich", 'sans-serif'].join(",")
}

export const theme = createTheme({
    palette,
    typography: {
        allVariants: {
            color: palette.text.default,
            fontFamily: ["Ubuntu", 'sans-serif'].join(",")
        },
        h1: { ...headingStyles },
        h2: { ...headingStyles },
        h3: { ...headingStyles },
        h4: { ...headingStyles },
        h5: { ...headingStyles }
    },
    shape: {
        borderRadius: "20px"
    },
    components: {
        MuiButton: {
            styleOverrides: {
                root: {
                    padding: "8px 16px"
                }
            }
        },
        // MuiPaper: {
        //     styleOverrides: {
        //         elevation1: {
        //             // boxShadow: `0 0 8px ${palette.text.dark}`
        //         }
        //     }
        // }
    }
})