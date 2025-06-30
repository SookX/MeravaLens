import { Button, Card, Checkbox, Divider, FormControlLabel, Stack, styled, TextField, Typography } from "@mui/material"
import { theme } from "../../theme/theme"
import { Link } from "react-router-dom"

const AccountForm = ({ 
    title = "",
    text = "",
    error = null,
    inputs = [],
    forgotPassword = false,
    rememberMeRef = null,
    handleSubmit = () => {},
    buttonLabel = "Submit",
    link = null,
    oauth = null
}) => {
    const FormCard = styled(Card)(({ theme })=>({
        padding: theme.spacing(6),
        width: theme.spacing(80),
        textAlign: "center",

        backgroundColor: "transparent"
    }))

    const FormSection = styled(Stack)(({ theme })=>({
        alignItems: "center",
        justifyContent: "center",
        padding: `${theme.spacing(16)} 0`,
    }))

    const ForgotPassword = styled(Typography)(({ theme })=>({
        paddingRight: theme.spacing(1),
        textDecoration: "underline",
        textAlign: "end"
    }))

    const StyledDivider = styled(Divider)(({ theme })=>({
        margin: `${theme.spacing(3)} 0`
    }))

    return (
        <FormSection>
            <FormCard>
                <Stack mb={4}>
                    <Typography variant="h3" color="primary">{title}</Typography>
                    <Typography variant="body1">{text}</Typography>
                    { error && <Typography variant="body1" color="error">{error}</Typography> }
                </Stack>

                <Stack gap={2}>
                    {
                        inputs.map((input, i) => (
                            <TextField
                                key={i}
                                type={input.type}
                                variant="outlined"
                                label={input.label}
                                inputRef={input.ref}
                            />
                        ))
                    }
                </Stack>

                <Stack mt={2} direction="row" alignItems="center" justifyContent="space-between">
                    {
                        rememberMeRef !== null &&
                        <FormControlLabel 
                            control={
                                <Checkbox color="primary" inputRef={rememberMeRef} />
                            } 
                            label="Remember me" 
                        />
                    }

                    {
                        forgotPassword &&
                        <Link to="/forgot-password"><ForgotPassword variant="body1" color="primary">Forgot Password</ForgotPassword></Link>
                    }
                </Stack>

                <Button sx={{ marginTop: 2 }} fullWidth size="large" variant="contained" onClick={handleSubmit}>{buttonLabel}</Button>

                {
                    oauth &&
                    <>
                        <StyledDivider>
                            <Typography variant="body1">Or</Typography>
                        </StyledDivider>
                        {oauth.component}
                    </>
                }
                
                {
                    link &&
                    <Stack mt={2} justifyContent="center" direction="row" gap={1}>
                        <Typography variant="body2">{link.text}</Typography>
                        <Link to={link.link}><Typography color="primary" fontWeight="bold" variant="body2">{link.label}</Typography></Link>
                    </Stack>
                }
            </FormCard>
        </FormSection>
    )
}

export default AccountForm