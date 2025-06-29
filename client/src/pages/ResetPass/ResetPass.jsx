import { Box, Button, Divider, Stack, styled, TextField, Typography } from "@mui/material"
import { useContext, useEffect, useRef, useState } from "react"
import { Link, useParams } from "react-router-dom"
import { DataContext } from "../../context/DataContext"
import { theme } from "../../theme/theme"
import { CheckCircleOutline, HighlightOff, Password } from "@mui/icons-material"

const ResetPass = () => {
    // Gets the url params
    const { uidb64, token } = useParams()



    // Gets global data from the context
    const { crud } = useContext(DataContext)



    // Holds the error state
    const [error, setError] = useState(null)
    const passwordRef = useRef()



    // Sends an reset password request to the backend on init
    const handleResetPassword = async () => {
        const response = await crud({
            url: `/users/reset-password/${uidb64}/${token}/`,
            method: "post",
            body: {
                password: passwordRef.current.value
            }
        })

        if(response.status !== 200) setError(response.response.data.error)
        console.log(response)
    }



    return (
        <>
            <TextField
                type="password"
                inputRef={passwordRef}
            />
            <Button onClick={handleResetPassword}>Reset</Button>
        </>
    )
}

export default ResetPass