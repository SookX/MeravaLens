import { Button, Checkbox, FormControlLabel, Stack, TextField, Typography } from "@mui/material"
import { Link } from "react-router-dom"
import { ForgotPassword, FormCard, FormSection, StyledDivider } from "./styling"

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
    
    return (
        <FormSection>
            <FormCard>
                    <Stack mb={{ xs: 3, md: 4 }}>
                        <Typography variant="h3" color="primary">{title}</Typography>
                        <Typography variant="body1">{text}</Typography>
                        { error && <Typography variant="body1" color="error">{error}</Typography> }
                    </Stack>

                <form onSubmit={handleSubmit}>
                    <Stack gap={{ xs: 1, md: 2 }}>
                        {
                            inputs.map((input, i) => (
                                <TextField
                                    key={i}
                                    type={input.type}
                                    variant="outlined"
                                    label={input.label}
                                    inputRef={input.ref}
                                    autoComplete={input.label}
                                />
                            ))
                        }
                    </Stack>

                    <Stack mt={{ xs: 1, md: 2 }} direction="row" alignItems="center" justifyContent="space-between">
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

                    <Button type="submit" sx={{ marginTop: 2 }} fullWidth size="large" variant="contained">{buttonLabel}</Button>
                </form>
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