import { Card, styled } from "@mui/material";

export const DialogBox = styled(Card)(({ theme })=>({
    padding: `${theme.spacing(4)} ${theme.spacing(3)}`,
    textAlign: "center"
}))