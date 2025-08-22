import { Button, Dialog, Typography } from "@mui/material"
import { Link } from "react-router-dom"
import { DialogBox } from "./styling"

const AuthenticationMessage = ({ 
    open = false,
    onClose = () => {},
    title = "",
    email = "",
    message = "",
    text = ""
}) => {

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