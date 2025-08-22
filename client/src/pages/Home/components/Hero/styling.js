import { Box, Divider, styled } from "@mui/material"

export const TextBox = styled(Box)(({ theme })=>({
    padding: `${theme.spacing(16)} ${theme.spacing(16)}`,
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    [theme.breakpoints.down("md")]: {
        alignItems: "center",
        textAlign: "center",
        padding: `${theme.spacing(16)} ${theme.spacing(8)}`,
        paddingBottom: 0
    },
    [theme.breakpoints.down("sm")]: {
        padding: `${theme.spacing(16)} ${theme.spacing(4)}`,
        paddingBottom: 0
    },
    height: "100%",
    position: "relative"
}))

export const Welcome = styled(Box)(({ theme })=>({
    position: "relative"
}))

export const StyledDivider = styled(Divider)(({ theme })=>({
    position: "absolute",
    width: theme.spacing(8),
    top: "50%",
    transfrom: "translateY(-50%)",
    backgroundColor: theme.palette.text.default
}))

export const LeftDivider = styled(StyledDivider)(({ theme })=>({
    left: `-${theme.spacing(9)}`,
}))

export const RightDivider = styled(StyledDivider)(({ theme })=>({
    [theme.breakpoints.up("md")]: {
        display: "none"
    },
    right: theme.spacing(-9)
}))

export const Circle = styled(Box)(({ theme })=>({
    width: theme.spacing(80),
    aspectRatio: "1 / 1",
    border: `solid 1px ${theme.palette.text.dark}`,
    borderRadius: "100%",
    position: "absolute",
    left: "-30%",
    top: "50%",
    zIndex: 0,

    [theme.breakpoints.down("md")]: { display: "none" }
}))