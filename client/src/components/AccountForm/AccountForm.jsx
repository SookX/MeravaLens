import { Box, Button, Card, Grid, InputAdornment, Stack, styled, TextField, Typography } from "@mui/material"
import { theme } from "../../theme/theme"
import satellite from "../../img/satellite.webp"
import { Link } from "react-router-dom"

const AccountForm = ({ title = "", text = "", error = null, inputs = [], handleForgotPassword = null, handleSubmit = () => {}, link = null }) => {
    const Section = styled(Grid)(({ theme })=>({
        minHeight: "100vh"
    }))

    const FormCard = styled(Card)(({ theme })=>({
        padding: theme.spacing(6),
        width: theme.spacing(80),
        textAlign: "center",

        backgroundColor: "transparent"
    }))

    const FormSection = styled(Stack)(({ theme })=>({
        alignItems: "center",
        justifyContent: "center",
        height: "100%"
    }))

    const Image = styled(Box)({
        height: "100%",
        backgroundImage: `url(${satellite})`,
        backgroundSize: "cover"
    })

    const ForgotPassword = styled(Typography)(({ theme })=>({
        marginTop: theme.spacing(2),
        paddingRight: theme.spacing(1),
        textDecoration: "underline",
        textAlign: "end"
    }))

    return (
        <Section container>
            <Grid size={5}>
                <Image />
            </Grid>
            <Grid size="grow">
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

                        {
                            handleForgotPassword &&
                            <ForgotPassword onClick={handleForgotPassword} variant="body1" color="primary">Forgot Password</ForgotPassword>
                        }

                        <Button sx={{ marginTop: 2 }} fullWidth size="large" variant="contained" onClick={handleSubmit}>Submit</Button>

                        {
                            link &&
                            <Stack mt={2} justifyContent="center" direction="row" gap={1}>
                                <Typography variant="body2">{link.text}</Typography>
                                <Link to={link.link}><Typography color="primary" fontWeight="bold" variant="body2">{link.label}</Typography></Link>
                            </Stack>
                        }
                    </FormCard>
                </FormSection>
            </Grid>
        </Section>
    )
}

export default AccountForm