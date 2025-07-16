import { Button, Card, Dialog, styled, Typography } from "@mui/material"
import { theme } from "../../theme/theme"
import { MailOutline } from "@mui/icons-material"
import { Link } from "react-router-dom"

const AuthenticationMessage = ({ 
    open = false,
    onClose = () => {},
    title = "",
    email = "",
    message = "",
    text = ""
}) => {
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
                <Typography variant="h3" color="primary" mb={2}>{title}</Typography>
                <Typography variant="body1">{message}</Typography>

                <Typography mb={1} size={"large"} variant="body1">{text}</Typography>
                <Link to='/login'><Button variant="contained">Log In</Button></Link>
            </DialogBox>
        </Dialog>
    )
}

export default AuthenticationMessage