import { Box, Grid, Stack, styled } from "@mui/material"

export const Section = styled(Grid)(({ theme }) => ({
    padding: `${theme.spacing(8)} ${theme.spacing(8)}`,
    [theme.breakpoints.down("md")]: {
        textAlign: "center",
        padding: `${theme.spacing(8)} ${theme.spacing(4)}`,
    },
    [theme.breakpoints.down("sm")]: {
        padding: `${theme.spacing(8)} ${theme.spacing(2)}`,
    }
}))



export const ImageContainer = styled(Stack)(({ theme }) => ({
    position: "relative",
    height: "100%",
    flexDirection: "row",
    gap: theme.spacing(4)
}))



export const ImageBox = styled(Box)(({ theme }) => ({
    width: theme.spacing(40),
    position: "absolute",
    [theme.breakpoints.down("lg")]: {
        width: theme.spacing(35),
        position: "static",
        flex: 1
    },
}))