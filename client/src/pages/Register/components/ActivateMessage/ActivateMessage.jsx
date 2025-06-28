import { Button, Card, Dialog, Stack, styled, Typography } from "@mui/material"
import { theme } from "../../../../theme/theme"
import { MailOutline } from "@mui/icons-material"
import { Link } from "react-router-dom"

const ActivateMessage = ({ open = false, onClose = () => {}, email = "" }) => {
    const DialogBox = styled(Card)(({ theme })=>({
        padding: `${theme.spacing(4)} ${theme.spacing(3)}`,
        textAlign: "center"
    }))

    const MailIcon = styled(MailOutline)(({ theme })=>({
        width: theme.spacing(3),
        height: theme.spacing(3)
    }))

    return (
        <Dialog
            open={open}
            onClose={onClose}
        >
            <DialogBox>
                {/* <MailIcon color="primary" /> */}
                <Typography variant="h3" color="primary" mb={2}>Thanks for signing up!</Typography>
                <Typography variant="body1">We've sent an activation email to:</Typography>
                <Stack mb={4} direction="row" gap={1} alignItems="center" justifyContent="center">
                    <MailIcon color="primary" />
                    <Typography variant="body1" fontWeight="bold">{email}</Typography>
                </Stack>

                <Typography mb={1} size={"large"} variant="body1">Once you're done activating your account, you can</Typography>
                <Link to='/login'><Button variant="contained">Log In</Button></Link>
            </DialogBox>
        </Dialog>
    )
}

export default ActivateMessage