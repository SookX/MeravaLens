import { Box, Stack, styled } from "@mui/material"

export const StyledFooter = styled(Stack)(({ theme }) => ({
    backgroundColor: theme.palette.background.mid,
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
    textAlign: "center",

    padding: `${theme.spacing(5)} ${theme.spacing(1)}`
}))


export const IconContainer = styled(Stack)(({ theme }) => ({
    position: "absolute",
    [theme.breakpoints.down("md")]: {
        position: "static",
        flexDirection: "row",
        gap: theme.spacing(1),
        marginBottom: theme.spacing(2)
    },

    top: 0,
    left: theme.spacing(4),
    alignItems: "center",
}))


export const StyledDivider = styled(Box)(({ theme }) => ({
    borderRight: `solid 1px ${theme.palette.primary.dark}`,
    height: theme.spacing(2),

    [theme.breakpoints.down("md")]: {
        display: "none"
    }
}))