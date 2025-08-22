import { Box, Card, styled } from "@mui/material"

export const StyledCard = styled(Card)(({ theme })=>({
    padding: `${theme.spacing(4)} ${theme.spacing(4)}`,
    height: "100%"
}))



export const ImageContainer = styled(Box)(({ theme })=>({
    borderRadius: theme.shape.borderRadius,
    overflow: "hidden"
}))