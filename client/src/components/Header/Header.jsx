import { AppBar, Box, Button, Card, Dialog, DialogContent, List, Stack, styled, Toolbar, Typography } from "@mui/material"
import logo from "../../img/logo.webp"
import { Link } from "react-router-dom"
import { useContext, useEffect, useState } from "react"
import { DataContext } from "../../context/DataContext"
import { ArrowDropDown } from "@mui/icons-material"
import { theme } from "../../theme/theme"

const Header = () => {
    // Gets global data from the context
    const { access, setAccess, setRefresh, navigate, crud, setLoading } = useContext(DataContext)



    // Holds the state for the user
    const [user, setUser] = useState(null)



    // Gets user data on init
    useEffect(() => {
        const fetching = async () => {
            setLoading(true)

            const response = await crud({
                url: '/users/me/',
                method: 'get'
            })

            if(response.status == 200) setUser(response.data)

            setLoading(false)
        }

        fetching()
    }, [access])



    // Holds the state for the dropdown
    const [open, setOpen] = useState(false)



    // Deletes user info and tokens
    const handleLogOut = () => {
        setUser(null)
        localStorage.removeItem('access')
        sessionStorage.removeItem('access')
        localStorage.removeItem('refresh')
        sessionStorage.removeItem('refresh')
        setAccess(null)
        setRefresh(null)
        navigate('/')
    }



    // Holds the state for deleting user
    const [deleteUser, setDeleteUser] = useState(false)



    // Makes an account deletion request to the backend
    const handleDeleteUser = async () => {
        setLoading(true)

        const response = await crud({
            url: "/users/me/",
            method: "delete"
        })

        if(response.status == 200) {
            setDeleteUser(false)
            handleLogOut()
        }

        setLoading(false)
    }



    const UserBox = styled(Stack)(({ theme })=>({
        flexDirection: "row",
        gap: theme.spacing(1),
        alignItems: "center",
        cursor: "pointer",
        position: "relative"
    }))


    const Dropdown = styled(Box)(({ theme })=>({
        position: "absolute",
        bottom: 0,
        transform: "translateY(100%)",
        width: "100%",
        paddingTop: theme.spacing(1),
        cursor: "default"
    }))


    const DropdownCard = styled(Card)(({ theme })=>({
        padding: theme.spacing(1),
        textAlign: "center"
    }))


    const StyledDialogContent = styled(DialogContent)(({ theme })=>({
        textAlign: "center",
    }))



    return (
        <AppBar color="transparent">
            <Dialog open={deleteUser} onClose={() => setDeleteUser(false)}>
                <StyledDialogContent>
                    <Typography color="primary" variant="h3">Are you sure?</Typography>
                    <Typography mb={3} variant="body1">Are you sure you want to delete your account? This action is irrevertable.</Typography>
                    <Stack direction="row" gap={1} justifyContent="center">
                        <Button size={"large"} onClick={() => setDeleteUser(false)} color="primary" variant="outlined">Cancel</Button>
                        <Button size={"large"} onClick={handleDeleteUser} color="primary" variant="contained">Delete</Button>
                    </Stack>
                </StyledDialogContent>
            </Dialog>
            <Toolbar disableGutters justifyContent="space-between">
                <Box sx={{ flexGrow: 1 }}>
                    <Link to='/'>
                        <img className="nav-logo" src={logo} alt="Merava Lens logo" />
                    </Link>
                </Box>
                {
                    user ?
                    <UserBox onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
                        <Stack textAlign="end">
                            <Typography mb={-0.7} variant="body1" color="primary">{user.username}</Typography>
                            <Typography variant="body2" color="text.secondary">{user.email}</Typography>

                            {
                                open &&
                                <Dropdown>
                                    <DropdownCard>
                                        <List gap={1}>
                                            <Link to='/dashboard'><Typography mb={1} variant="body1">Dashboard</Typography></Link>
                                            <Link to='/change-password'><Typography mb={1} variant="body1">Change password</Typography></Link>
                                            <Typography sx={{ cursor: "pointer" }} onClick={() => setDeleteUser(true)} mb={1} variant="body1" color="warning">Delete account</Typography>
                                            <Typography sx={{ cursor: "pointer" }} onClick={handleLogOut} variant="body1" color="error">Log out</Typography>
                                        </List>
                                    </DropdownCard>
                                </Dropdown>
                            }
                        </Stack>
                        <ArrowDropDown sx={{ transform: `rotate(${open ? "180deg" : "0"})` }} color="text" />
                    </UserBox>
                    :
                    <Link to='/register'><Button variant="text" color="text">Get started</Button></Link>
                }
            </Toolbar>
        </AppBar>
    )
}

export default Header