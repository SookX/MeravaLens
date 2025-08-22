import { Card, Divider, Stack, styled, Typography } from "@mui/material"

export const FormCard = styled(Card)(({ theme })=>({
    padding: theme.spacing(6),
    [theme.breakpoints.down("lg")]: { padding: theme.spacing(5) },
    [theme.breakpoints.down("md")]: { padding: theme.spacing(6) },
    [theme.breakpoints.down("sm")]: { padding: theme.spacing(3) },

    width: "100%",
    textAlign: "center",

    backgroundColor: "transparent"
}))

export const FormSection = styled(Stack)(({ theme })=>({
    alignItems: "center",
    justifyContent: "center",
    padding: `${theme.spacing(16)} ${theme.spacing(18)}`,
    [theme.breakpoints.down("lg")]: { padding: `${theme.spacing(16)} ${theme.spacing(12)}`, },
    [theme.breakpoints.down("md")]: { padding: `${theme.spacing(12)} ${theme.spacing(8)}`, },
    [theme.breakpoints.down("sm")]: { padding: `${theme.spacing(16)} ${theme.spacing(4)}`, }
}))

export const ForgotPassword = styled(Typography)(({ theme })=>({
    paddingRight: theme.spacing(1),
    textDecoration: "underline",
    textAlign: "end"
}))

export const StyledDivider = styled(Divider)(({ theme })=>({
    margin: `${theme.spacing(3)} 0`
}))