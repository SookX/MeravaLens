import { HelpOutline } from "@mui/icons-material"
import { Card, styled } from "@mui/material"

export const StyledCard = styled(Card)(({ theme }) => ({
    padding: theme.spacing(3)
}))



export const Icon = styled(HelpOutline)(({ theme })=>({
    color: theme.palette.primary.main,
    cursor: "pointer",
    width: theme.spacing(2),
    height: theme.spacing(2)
}))