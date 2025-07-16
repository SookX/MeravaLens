import { Box, Button, Divider, Stack, styled, TextField, Typography } from "@mui/material"
import { useContext, useEffect, useRef, useState } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { DataContext } from "../../context/DataContext"
import { theme } from "../../theme/theme"
import { CheckCircleOutline, HighlightOff, Password } from "@mui/icons-material"
import FormPage from "../../components/FormPage/FormPage"
import AccountForm from "../../components/AccountForm/AccountForm"
import { crud } from "../../api/crud"

const ResetPass = () => {
    // Gets the url params
    const { uidb64, token } = useParams()



    // Gets global data from the context
    const { setLoading } = useContext(DataContext)



    // Holds the error state
    const [error, setError] = useState(null)
    const passwordRef = useRef()

    const inputs = [
        {
            type: "password",
            label: "New Password",
            ref: passwordRef
        },
    ]



    // Sends an reset password request to the backend on init
    const handleResetPassword = async () => {
        setLoading(true)

        const response = await crud({
            url: `/users/reset-password/${uidb64}/${token}/`,
            method: "post",
            body: {
                password: passwordRef.current.value
            }
        })

        console.log(response)

        
        if(response.status == 200) useNavigate()('/login')
        else setError(response.response.data.error)
    
        setLoading(false)
    }



    return (
        <FormPage>
            <AccountForm
                title="Reset your password"
                text="You can reset your password now. Type the new password for your account below."
                error={error}
                inputs={inputs}
                handleSubmit={handleResetPassword}
                buttonLabel="Reset my password"
                link={{
                    text: "Done with setting your password?",
                    label: "Log in",
                    link: "/login"
                }}
            />
        </FormPage>
    )
}

export default ResetPass