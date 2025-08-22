import { Box, Divider, styled } from "@mui/material"

export const mapHeight = 500

export const Section = styled(Box)(({ theme })=>({
    padding: `${theme.spacing(8)} ${theme.spacing(24)}`,
    [theme.breakpoints.down("lg")]: { padding: `${theme.spacing(8)} ${theme.spacing(8)}` },
    [theme.breakpoints.down("sm")]: { padding: `${theme.spacing(8)} ${theme.spacing(2)}` }
}))



export const StyledDivider = styled(Divider)(({ theme })=>({
    "&::before, &::after": {
        borderColor: theme.palette.text.default
    }
}))



export const StyledContainer = styled(Box)(({ theme })=>({
    width: "75%",
    [theme.breakpoints.down("md")]: { width: "100%" },
    margin: "0 auto",
    position: "relative",
    marginBottom: `${mapHeight + 100}px`
}))