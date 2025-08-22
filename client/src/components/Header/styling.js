import { Box, Card, DialogContent, Stack, styled } from "@mui/material"

export const UserBox = styled(Stack)(({ theme })=>({
    flexDirection: "row",
    gap: theme.spacing(1),
    alignItems: "center",
    cursor: "pointer",
    position: "relative"
}))


export const Dropdown = styled(Box)(({ theme })=>({
    position: "absolute",
    bottom: 0,
    transform: "translateY(100%)",
    width: "100%",
    paddingTop: theme.spacing(1),
    cursor: "default"
}))


export const DropdownCard = styled(Card)(({ theme })=>({
    padding: theme.spacing(1),
    textAlign: "center"
}))


export const StyledDialogContent = styled(DialogContent)(({ theme })=>({
    textAlign: "center",
}))