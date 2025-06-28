import { useContext, useState } from "react"
import AccountForm from "../../components/AccountForm/AccountForm"
import { DataContext } from "../../context/DataContext"

const Register = () => {
    // Gets global data from the context
    const { crud } = useContext(DataContext)

    const [email, setEmail] = useState("")
    const [username, setUsername] = useState("")
    const [password, setPassword] = useState("")
    const [confirmPassword, setConfirmPassword] = useState("")

    const inputs = [
        {
            type: "email",
            label: "Email",
            value: email,
            setValue: setEmail
        },
        {
            type: "text",
            label: "Username",
            value: username,
            setValue: setUsername
        },
        {
            type: "password",
            label: "Password",
            value: password,
            setValue: setPassword
        },
        {
            type: "password",
            label: "Confirm Password",
            value: confirmPassword,
            setValue: setConfirmPassword
        },
    ]

    const handleSubmit = async () => {
        const response = await crud({
            url: "/register",
            method: "post",
            body: {
                email,
                username,
                password,
                confirmPassword
            }
        })

        console.log(response)
    }

    return (
        <AccountForm inputs={inputs} handleSubmit={handleSubmit} />
    )
}

export default Register