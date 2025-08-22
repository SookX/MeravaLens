import { Box, Divider, styled } from "@mui/material"

export const StyledBox = styled(Box)(({ theme })=>({
    padding: `${theme.spacing(8)} ${theme.spacing(16)}`,
    [theme.breakpoints.down("lg")]: { padding: `${theme.spacing(8)} ${theme.spacing(8)}` },
    [theme.breakpoints.down("sm")]: { padding: `${theme.spacing(8)} ${theme.spacing(2)}` }
}))



export const StyledDivider = styled(Divider)(({ theme })=>({
    "&::before, &::after": {
        borderColor: theme.palette.text.default
    }
}))