import { AppBar, Box, Button, Toolbar, Typography } from "@mui/material"
import logo from "../../img/logo.webp"
import { Link } from "react-router-dom"
import { useContext } from "react"
import { DataContext } from "../../context/DataContext"

const Header = () => {
    // Gets global data from the context
    const { access } = useContext(DataContext)



    return (
        <AppBar color="transparent">
            <Toolbar disableGutters justifyContent="space-between">
                <Box sx={{ flexGrow: 1 }}>
                    <Link to='/'>
                        <img className="nav-logo" src={logo} alt="Merava Lens logo" />
                    </Link>
                </Box>
                {
                    access ?
                    null
                    :
                    <Link to='/register'><Button variant="text" color="text">Get started</Button></Link>
                }
            </Toolbar>
        </AppBar>
    )
}

export default Header