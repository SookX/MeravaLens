import { createTheme } from "@mui/material"

const palette = {
    mode: "dark",
    background: {
        default: "#000",
        mid: "#121212",
        light: "#333333"
    },
    text: {
        main: "#DDE6ED",
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
        MuiAppBar: {
            styleOverrides: {
                root: {
                    boxShadow: "none"
                },
                colorDefault: {
                    backgroundColor: palette.background.default
                }
            }
        },
        MuiLink: {
            styleOverrides: {
                root: {
                    textDecoration: "none"
                }
            }
        }
    }
})

theme.components.MuiToolbar = {
    styleOverrides: {
        root: {
            paddingLeft: theme.spacing(8),
            paddingRight: theme.spacing(8),

            [theme.breakpoints.down("md")]: {
                paddingLeft: theme.spacing(4),
                paddingRight: theme.spacing(4)
            }
        }
    }
}

theme.typography.h1 = {
    ...theme.typography.h1,
    [theme.breakpoints.down("lg")]: { fontSize: theme.spacing(9) },
    [theme.breakpoints.down("md")]: { fontSize: theme.spacing(8) }
}

theme.typography.h2 = {
    ...theme.typography.h2,
    [theme.breakpoints.down("lg")]: { fontSize: theme.spacing(7) },
    [theme.breakpoints.down("md")]: { fontSize: theme.spacing(6) }
}

theme.typography.h3 = {
    ...theme.typography.h3,
    [theme.breakpoints.down("lg")]: { fontSize: theme.spacing(5) },
    [theme.breakpoints.down("sm")]: { fontSize: theme.spacing(4) }
}

theme.typography.h4 = {
    ...theme.typography.h4,
    [theme.breakpoints.down("lg")]: { fontSize: theme.spacing(3) }
}