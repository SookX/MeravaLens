import { CheckCircleOutline, HighlightOff } from "@mui/icons-material"
import { Box, Divider, Stack, styled } from "@mui/material"

export const Section = styled(Stack)(({theme})=>({
    padding: `${theme.spacing(20)} ${theme.spacing(25)}`,
    [theme.breakpoints.down("md")]: { padding: `${theme.spacing(16)} ${theme.spacing(12)}` },
    [theme.breakpoints.down("sm")]: { padding: `${theme.spacing(12)} ${theme.spacing(4)}` },

    textAlign: "center",
    alignItems: "center",
    justifyContent: "center",
    minHeight: "100vh",
    zIndex: 1
}))


export const IconSuccess = styled(CheckCircleOutline)(({ theme })=>({
    width: theme.spacing(20),
    height: theme.spacing(20),
    
    [theme.breakpoints.down("lg")]: {
        width: theme.spacing(16),
        height: theme.spacing(16)
    },

    [theme.breakpoints.down("sm")]: {
        width: theme.spacing(12),
        height: theme.spacing(12)
    },

    marginBottom: theme.spacing(2)
}))


export const IconError = styled(HighlightOff)(({ theme })=>({
    width: theme.spacing(20),
    height: theme.spacing(20),

    [theme.breakpoints.down("lg")]: {
        width: theme.spacing(16),
        height: theme.spacing(16)
    },

    [theme.breakpoints.down("sm")]: {
        width: theme.spacing(12),
        height: theme.spacing(12)
    },

    marginBottom: theme.spacing(2)
}))



export const Circle = styled(Box)(({ theme })=>({
    aspectRatio: "1 / 1",
    border: `solid 1px ${theme.palette.text.dark}`,
    borderRadius: "100%",
    position: "absolute",
    zIndex: 0
}))


export const StyledDivider = styled(Divider)(({ theme })=>({
    width: theme.spacing(6),
    background: theme.palette.primary.main
}))