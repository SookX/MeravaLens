import { Box, Button, Divider, Stack, styled, Typography } from "@mui/material"
import { useContext, useEffect, useState } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { DataContext } from "../../context/DataContext"
import { theme } from "../../theme/theme"
import { CheckCircleOutline, HighlightOff } from "@mui/icons-material"
import { crud } from "../../api/crud"

const Activate = () => {
    // Gets the url params
    const { uidb64, token } = useParams()



    // Gets global data from the context
    const { access, setLoading } = useContext(DataContext)



    // Checks if the user is already authenticated
    useEffect(() => {
        if(access) useNavigate()('/dashboard')
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
            console.log(response)

            setLoading(false)
        }

        activate()
    }, [])



    const Section = styled(Stack)(({theme})=>({
        padding: `${theme.spacing(20)} ${theme.spacing(25)}`,
        [theme.breakpoints.down("md")]: { padding: `${theme.spacing(16)} ${theme.spacing(12)}` },
        [theme.breakpoints.down("sm")]: { padding: `${theme.spacing(12)} ${theme.spacing(4)}` },

        textAlign: "center",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        zIndex: 1
    }))


    const IconSuccess = styled(CheckCircleOutline)(({ theme })=>({
        width: theme.spacing(20),
        height: theme.spacing(20),
        
        [theme.breakpoints.down("lg")]: {
            width: theme.spacing(16),
            height: theme.spacing(16)
        },

        [theme.breakpoints.down("sm")]: {
            width: theme.spacing(12),
            height: theme.spacing(12)
        },

        marginBottom: theme.spacing(2)
    }))


    const IconError = styled(HighlightOff)(({ theme })=>({
        width: theme.spacing(20),
        height: theme.spacing(20),

        [theme.breakpoints.down("lg")]: {
            width: theme.spacing(16),
            height: theme.spacing(16)
        },

        [theme.breakpoints.down("sm")]: {
            width: theme.spacing(12),
            height: theme.spacing(12)
        },

        marginBottom: theme.spacing(2)
    }))



    const Circle = styled(Box)(({ theme })=>({
        aspectRatio: "1 / 1",
        border: `solid 1px ${theme.palette.text.dark}`,
        borderRadius: "100%",
        position: "absolute",
        zIndex: 0
    }))


    const StyledDivider = styled(Divider)(({ theme })=>({
        width: theme.spacing(6),
        background: theme.palette.primary.main
    }))



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