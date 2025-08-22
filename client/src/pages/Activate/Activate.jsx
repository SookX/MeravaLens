import { Box, Button, Stack, Typography } from "@mui/material"
import { useContext, useEffect, useState } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { DataContext } from "../../context/DataContext"
import { theme } from "../../theme/theme"
import { crud } from "../../api/crud"
import { Circle, IconError, IconSuccess, Section, StyledDivider } from "./styling"

const Activate = () => {
    // Gets the url params
    const { uidb64, token } = useParams()



    // Gets global data from the context
    const { access, setLoading } = useContext(DataContext)



    // Navigates users to another page
    const navigate = useNavigate()



    // Checks if the user is already authenticated
    useEffect(() => {
        if(access) navigate('/dashboard')
    }, [access])



    // Holds the error state
    const [error, setError] = useState(null)



    // Sends an activation request to the backend on init
    useEffect(() => {
        const activate = async () => {
            setLoading(true)

            const response = await crud({
                url: `/users/activate/${uidb64}/${token}/`,
                method: "get"
            })

            if(response.status !== 200) setError(response.response.data.error)

            setLoading(false)
        }

        activate()
    }, [])



    return (
        <Box sx={{ position: "relative", overflow: "hidden" }}>
            <Circle sx={{ top: "50%", left: "-20%", width: theme.spacing(100) }} />
            <Circle sx={{ bottom: "55%", right: "-15%", width: theme.spacing(50) }} />
            <Circle sx={{ top: "-50%", left: "15%", width: theme.spacing(50) }} />
            <Section>
                {
                    error ?
                    <IconError color="primary" />
                    :
                    <IconSuccess color="primary" />
                }
                <Typography mb={1} variant="h3" color="primary">
                    {
                        error ?
                        'Error activating your account!'
                        :
                        'Your account is now active!'
                    }
                </Typography>
                {
                    error ?
                    <Typography variant="body1">{error}</Typography>
                    :
                    <Typography variant="body1">Thank you for registering a <Typography variant="span" color="primary">Merava Lens</Typography> account. You may now login and use all of our services freely.</Typography>
                }
                <Typography mb={3} variant="body1">
                    {
                        error ?
                        'Try again later.'
                        :
                        'We wish you the best experience!'
                    }
                </Typography>
                
                <Stack gap={1} direction="row" alignItems="center">
                    <StyledDivider />
                    <Link to={`${error ? '/' : '/login'}`}>
                        <Button variant="outlined">
                            {
                                error ?
                                'Home'
                                :
                                'Log in'
                            }
                        </Button>
                    </Link>
                    <StyledDivider />
                </Stack>
            
            </Section>
        </Box>
    )
}

export default Activate